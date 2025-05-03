#!/usr/bin/env python3
import os
import numpy as np
import torch
import logging  # Import the logging module

from PyISV.features_calc_utils import build_rdf

# Define global constants for testing purposes
XYZ_PATH = "./data/structures/full_min_ptmd_m18_nCu_0.xyz"
LABEL_FILE = "./data/Ag38_labels/combined_isv_labels_2D_nonMin_to_min_k15_nCu_0.txt"
OUTPUT_DIR = "./data/RDFs"
MIN_DIST = 1.0
MAX_DIST = 8.0
N_BINS = 200
BANDWIDTH = 0.2
FRACTION = 1.0

def setup_logging(log_file):
    """
    Setup the logging configuration.
    """
    logging.basicConfig(filename=log_file, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    return log_file


if __name__ == "__main__":
    # Set up logging
    log_file = setup_logging("./logs/rdf_computation.log")

    # Run the RDF computation
    build_rdf(
        xyz_path=XYZ_PATH,
        min_dist=MIN_DIST,
        max_dist=MAX_DIST,
        n_bins=N_BINS,
        bandwidth=BANDWIDTH,
        output_path=OUTPUT_DIR,
        fraction=FRACTION,
    )

    # Generate labels torch file
    label_file = LABEL_FILE
    output_path=OUTPUT_DIR

    labels = torch.LongTensor(np.loadtxt(label_file, dtype=int))
    label_path = os.path.join(output_path, "labels.pt")
    torch.save(labels, label_path)

    print(f"Log saved to: {log_file}")
    logging.info(f"Saved labels to {label_path}")

