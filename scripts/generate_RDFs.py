# This file generates radial distribution functions (RDFs) from a given XYZ file.

import torch
import os
import logging  # Import the logging module
import sys

# Common project paths
from PyISV import (
    MODELS_DIR as models_dir, DATA_DIR as data_dir, OUTPUTS_DIR as outputs_dir,
    NORMS_DIR as norms_dir, STATS_DIR as stats_dir, LOGS_DIR as logs_dir,
)
# Import the build_rdf function from the features_calc_utils module
from PyISV.features_calc_utils import build_rdf

# Define global constants for testing purposes
XYZ_PATH = f"{data_dir}/structures/full_min_ptmd_m18_nCu_0.xyz"
OUTPUT_DIR = f"{data_dir}/RDFs"
MIN_DIST = 1.0
MAX_DIST = 12.0
N_BINS = 340
BANDWIDTH = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_logging(log_file):
    """ Setup the logging configuration. """
    logging.basicConfig(filename=log_file, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    return log_file

if __name__ == "__main__":
    # Set up logging
    log_file = setup_logging(f"{logs_dir}/rdf_computation.log")
    logging.info("Starting RDF computation...")

    print("Starting RDF computation...")

    try:
        # Run the RDF computation
        build_rdf(
            xyz_path=XYZ_PATH,
            min_dist=MIN_DIST,
            max_dist=MAX_DIST,
            n_bins=N_BINS,
            bandwidth=BANDWIDTH,
            output_path=OUTPUT_DIR,
            device=DEVICE,
        )
    except Exception as e:
        logging.error(f"Error during RDF computation: {e}")
        print(f"Error during RDF computation: {e}")
    finally:
        print("RDF computation finished.")
        logging.info("RDF computation finished.")

