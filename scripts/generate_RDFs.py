#!/usr/bin/env python3
import os
import numpy as np
import torch
import logging  # Import the logging module

from PyISV.features_calc_utils import build_rdf

# Define global constants for testing purposes
XYZ_PATH = "./data/structures/full_min_ptmd_m18_nCu_0.xyz"
OUTPUT_DIR = "./data/RDFs"
MIN_DIST = 1.0
MAX_DIST = 8.0
N_BINS = 320
BANDWIDTH = 0.1

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
    )

