#!/usr/bin/env python3
import os
import time
import torch
import numpy as np
from ase.io import read
from tqdm import tqdm
import logging  # Import the logging module

from PyISV.features_calc_utils import single_kde_calc


def setup_logging(output_path):
    """
    Setup the logging configuration.
    """
    log_file = os.path.join(output_path, "rdf_computation.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    return log_file

def compute_rdfs_and_save(xyz_path, label_file, min_dist, max_dist, n_bins, bandwidth,
                        output_path, fraction=1.0, periodic=False, replicate=(2,2,2)):
    """
    Computes RDFs and saves them as images.
    """

    # Log the input parameters
    logging.info("Starting RDF computation with the following parameters:")
    logging.info(f"xyz_path: {xyz_path}")
    logging.info(f"label_file: {label_file}")
    logging.info(f"min_dist: {min_dist}, max_dist: {max_dist}, n_bins: {n_bins}, bandwidth: {bandwidth}")
    logging.info(f"fraction: {fraction}, periodic: {periodic}, replicate: {replicate}")
    
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

        # compute RDF directly from Atoms
        rdf_out = single_kde_calc(atoms, bins, bw, mic=periodic)

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
    

if __name__ == "__main__":
    # — user config —
    XYZ_PATH    = "./structures/full_min_ptmd_m18_nCu_0.xyz"
    LABEL_FILE  = "./Ag38_labels/combined_isv_labels_2D_nonMin_to_min_k15_nCu_0.txt"
    OUTPUT_DIR  = "./RDFs"

    MIN_DIST    = 1.0       # Minimum distance for RDF calculation (in Angstroms)
    MAX_DIST    = 8.0       # Maximum distance for RDF calculation (in Angstroms)
    N_BINS      = 200       # Number of bins for the histogram
    BANDWIDTH   = 0.2       # Bandwidth for KDE (in Angstroms)

    FRACTION    = 1.0       # Fraction of frames to use for RDF calculation (1.0 = all frames)
    PERIODIC    = False     # Set to True for periodic boundary conditions
    REPLICATE   = (2, 2, 2) # Optional, only used if periodic=True
    
    # Set up logging
    log_file = setup_logging(OUTPUT_DIR)

    # Run the RDF computation
    compute_rdfs_and_save(
        xyz_path=XYZ_PATH,
        label_file=LABEL_FILE,
        min_dist=MIN_DIST,
        max_dist=MAX_DIST,
        n_bins=N_BINS,
        bandwidth=BANDWIDTH,
        output_path=OUTPUT_DIR,
        fraction=FRACTION,
        periodic=PERIODIC,
        replicate=REPLICATE,
    )

    print(f"Log saved to: {log_file}")
