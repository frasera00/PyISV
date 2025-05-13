# This file generates radial distribution functions (RDFs) from a given XYZ file.

from PyISV.features_calc_utils import build_rdf

import os
import numpy as np
import logging  # Import the logging module

# Get the absolute path to the PyISV root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYISV_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# Build paths relative to the PyISV root:
data_dir = os.path.join(PYISV_ROOT, 'data')

# Define global constants for testing purposes
XYZ_PATH = f"{data_dir}/structures/full_min_ptmd_m18_nCu_0.xyz"
OUTPUT_DIR = f"{data_dir}/RDFs"
MIN_DIST = 1.0
MAX_DIST = 12.0
N_BINS = 340
BANDWIDTH = 0.2

def setup_logging(log_file):
    """ Setup the logging configuration. """
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

