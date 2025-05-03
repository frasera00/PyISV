import numpy as np
from typing import Tuple
import torch
import logging
import os
import time
from tqdm import tqdm
from ase.io import read

#-----------------------------------------
# ASE-TORCH KDE RDF FUNCTIONS 
#-----------------------------------------

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
def normalized_histogram(x: torch.Tensor, bins: torch.Tensor, factor: torch.Tensor, bandwidth: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    if x.dim() == 1:
        x = x.unsqueeze(0)  # Ensure x has at least 2 dimensions
    pdf, _ = marginal_pdf(x.unsqueeze(2), bins, bandwidth, epsilon)
    return pdf * factor

def compute_ase_all_distances(Atoms, mic=False):
    distances = np.array(Atoms.get_all_distances(mic))  # Ensure distances is a NumPy array
    n_atoms = len(distances)
    indices = torch.triu_indices(n_atoms, n_atoms, offset=1)
    return torch.tensor(distances[indices[0], indices[1]])

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
    return torch_kde_calc(torch.Tensor(distances), bins, bandwidth)  # Return as a PyTorch tensor

def compute_rdf(atoms, bins, bandwidth, mic=False, mode="single", specie1=None, specie2=None):
    """
    Compute the Radial Distribution Function (RDF) or Triple KDE for the given atomic configuration.

    Parameters:
    - atoms: ASE Atoms object.
    - bins: Torch tensor representing the bins for the KDE.
    - bandwidth: Torch tensor representing the bandwidth for the KDE.
    - mic: Boolean indicating whether to use minimum image convention.
    - mode: "single" for single RDF or "triple" for triple KDE.
    - specie1: First species for triple KDE (required if mode="triple").
    - specie2: Second species for triple KDE (required if mode="triple").

    Returns:
    - RDF or Triple KDE as a Torch tensor or tuple of tensors.
    """
    if mode == "single":
        distances = compute_ase_all_distances(atoms, mic=mic)
        return torch_kde_calc(torch.Tensor(distances), bins, bandwidth)

    elif mode == "triple":
        if specie1 is None or specie2 is None:
            raise ValueError("specie1 and specie2 must be provided for triple KDE.")
        return triple_kde_calc(specie1, specie2, atoms, bins, bandwidth, mic=mic)

    else:
        raise ValueError("Invalid mode. Choose 'single' or 'triple'.")

def build_rdf(xyz_path, label_file, min_dist, max_dist, n_bins, bandwidth,
              output_path, fraction=1.0, periodic=False, replicate=(2,2,2),
              mode="single", specie1=None, specie2=None):
    """
    Builds RDFs (single or triple) and saves them as images.

    Parameters:
    - xyz_path: Path to the XYZ file containing atomic configurations.
    - label_file: Path to the label file.
    - min_dist: Minimum distance for RDF calculation (in Angstroms).
    - max_dist: Maximum distance for RDF calculation (in Angstroms).
    - n_bins: Number of bins for the histogram.
    - bandwidth: Bandwidth for KDE (in Angstroms).
    - output_path: Directory to save the RDF images and labels.
    - fraction: Fraction of frames to use for RDF calculation (1.0 = all frames).
    - periodic: Boolean indicating whether to use periodic boundary conditions.
    - replicate: Tuple specifying replication factors for periodic systems.
    - mode: "single" for single RDF or "triple" for triple KDE.
    - specie1: First species for triple KDE (required if mode="triple").
    - specie2: Second species for triple KDE (required if mode="triple").
    """

    # Log the input parameters
    logging.info("Starting RDF computation with the following parameters:")
    logging.info(f"xyz_path: {xyz_path}")
    logging.info(f"label_file: {label_file}")
    logging.info(f"min_dist: {min_dist}, max_dist: {max_dist}, n_bins: {n_bins}, bandwidth: {bandwidth}")
    logging.info(f"fraction: {fraction}, periodic: {periodic}, replicate: {replicate}, mode: {mode}")

    # 1) load frames & labels
    frames = read(xyz_path, index=":")
    N      = len(frames)
    labels = torch.LongTensor(np.loadtxt(label_file, dtype=int))
    assert labels.numel() == N

    # 2) prepare bins + bandwidth
    bins = torch.linspace(min_dist, max_dist, n_bins)
    bw   = torch.tensor([bandwidth], dtype=torch.float32)

    # 3) select subset of frames
    n_sub = int(N * fraction)
    perm  = torch.randperm(N)[:n_sub]

    # 4) allocate output tensor
    shape = [n_sub, 1] + [n_bins]
    images = torch.empty(shape, dtype=torch.float32)

    # 5) loop and compute
    t0 = time.time()
    for out_i, frame_i in enumerate(tqdm(perm, desc="Progress")):
        atoms = frames[frame_i]

        # optional periodic replication (more useful for bulk systems)
        if periodic:
            atoms.set_pbc(True)
            atoms = atoms.repeat(*replicate)

        # compute RDF or Triple KDE using compute_rdf
        rdf_out = compute_rdf(atoms, bins, bw, mic=periodic, mode=mode, specie1=specie1, specie2=specie2)

        # handle returns: could be np.ndarray or torch.Tensor
        if isinstance(rdf_out, np.ndarray):
            rdf = torch.from_numpy(rdf_out).float()
        else:
            rdf = rdf_out.cpu().float()

        # if shape is (1, num_bins), squeeze to (num_bins,)
        if rdf.ndimension() == 2 and rdf.size(0) == 1:
            rdf = rdf.squeeze(0)
        elif rdf.ndimension() != 1:
            raise ValueError(f"Unexpected RDF shape {rdf.shape}")

        # create RDF image
        images[out_i, 0] = rdf

    elapsed = time.time() - t0
    logging.info(f"Computed {n_sub}/{N} frames in {elapsed:.2f}s ({elapsed/n_sub:.3f}s each)")

    # 6) save images and labels
    os.makedirs(output_path, exist_ok=True)
    img_path   = os.path.join(output_path, f"rdf_images.pt")
    label_path = os.path.join(output_path, "labels.pt")
    torch.save(images, img_path)
    torch.save(labels[perm], label_path)
    
    logging.info(f"Saved RDF images to {img_path}")
    logging.info(f"Saved labels to {label_path}")
    logging.info("RDF computation completed.")

