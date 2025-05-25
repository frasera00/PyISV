from typing import Tuple
from ase.io import read
from tqdm import tqdm
import numpy as np
import time
import torch
import logging
import itertools


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
def torch_kde_calc(pos1: torch.Tensor, pos2: torch.Tensor, bins: torch.Tensor, bandwidth: torch.Tensor) -> torch.Tensor:
    """Calculate the kernel density estimation (KDE) for two sets of positions."""
    distances = torch.cdist(pos1, pos2)
    distances_flat = distances.reshape(-1)
    return normalized_histogram(distances_flat, bins, computefactor(bins), bandwidth)

def calc_rdfs(params: dict) -> Tuple[torch.Tensor, list]:
    """Calculate RDFs for all frames according to the specified mode"""
    print("Parameters:")
    for key, value in params.items():
        print(f"{key}: {value}")

    # Unpack parameters
    xyz_file = params['xyz_file']
    min_dist, max_dist = params['min_dist'], params['max_dist']
    n_bins, bandwidth = params['n_bins'], params['bandwidth']
    device = params['device']
    output_file = params['output_file']
    mode = params.get('mode', 'multi')
    species = params.get('species', ['unknown'])
    fraction = params.get('fraction', 1.0)

    # Read XYZ file
    frames = read(xyz_file, index=":")

    # Subset of frames for testing
    n_sub = int(len(frames) * fraction)
    subset = torch.randperm(len(frames))[:n_sub] if fraction < 1.0 else torch.arange(n_sub)

    # Set parameters for RDF calculation
    bins = torch.linspace(min_dist, max_dist, n_bins, device=device)
    bw = torch.tensor([bandwidth], dtype=torch.float32, device=device)
    
    # Define pairs of species (including self-pairs)
    pairs = list(itertools.combinations_with_replacement(set(species), 2))
    print(f"Pairs found: {pairs}")

    # Lists to track processing
    valid_frames = []
    skipped_frames = []
    all_rdfs = []
    
    t0 = time.time()

    for frame_idx in tqdm(subset, desc="Calculating RDF", unit="frame"):
        try:
            frame = frames[int(frame_idx.item())]

            # Get positions by element
            positions_by_element = {}
            symbols = frame.get_chemical_symbols() # type: ignore[assignment]
            atom_positions = frame.get_positions() # type: ignore[assignment]
            
            for element in species:
                indices = [i for i, s in enumerate(symbols) if s == element]
                if indices:  # Only add if the element exists in this frame
                    positions_by_element[element] = torch.tensor(
                        atom_positions[indices], dtype=torch.float32, device=device)
            
            # Check if all needed species are present
            if not all(s in positions_by_element for s in species):
                skipped_frames.append(frame_idx.item())
                continue
                
            # Calculate RDFs for all pairs
            pair_rdfs = []
            
            for pair in pairs:
                specie1, specie2 = pair
                pos1 = positions_by_element[specie1]
                pos2 = positions_by_element[specie2]
                
                # Calculate distances
                distances = torch.cdist(pos1, pos2)
                
                # Handle same-species case (exclude self-distances)
                if specie1 == specie2:
                    if len(pos1) > 1:  # Need at least 2 atoms for meaningful same-species RDF
                        mask = ~torch.eye(len(pos1), dtype=torch.bool, device=device)
                        distances_flat = distances[mask]
                    else:
                        # Single atom case - use empty tensor with proper shape
                        distances_flat = torch.zeros(0, device=device)
                else:
                    # Different species - use all distances
                    distances_flat = distances.flatten()
                
                # If we have distances, calculate histogram
                if len(distances_flat) > 0:
                    rdf = normalized_histogram(distances_flat, bins, computefactor(bins), bw)
                    pair_rdfs.append(rdf)
                else:
                    pair_rdfs.append(torch.zeros_like(bins).unsqueeze(0))
            
            # Format based on mode
            if mode == 'multi':
                # Stack along channel dimension - [n_pairs, n_bins]
                frame_rdf = torch.cat(pair_rdfs, dim=0)
            elif mode == 'concat':
                # Concatenate all pairs - [1, n_pairs*n_bins]
                frame_rdf = torch.cat([r.flatten() for r in pair_rdfs], dim=0).unsqueeze(0)
            elif mode == 'single':
                # Average across pairs - [1, n_bins]
                frame_rdf = torch.mean(torch.stack(pair_rdfs, dim=0), dim=0)
            
            all_rdfs.append(frame_rdf)
            valid_frames.append(frame_idx.item())
            
        except Exception as e:
            print(f"Error processing frame {frame_idx.item()}: {e}")
            skipped_frames.append(frame_idx.item())
    
    # Stack all frames
    if all_rdfs:
        rdf_tensor = torch.stack(all_rdfs, dim=0)
    else:
        raise ValueError("No valid frames found for RDF calculation.")
    
    # Save results
    torch.save(rdf_tensor, output_file)
    elapsed = time.time() - t0
    print(f"Computed {len(valid_frames)}/{len(subset)} valid frames in {elapsed:.2f}s")
    
    return rdf_tensor, skipped_frames