# This file generates radial distribution functions (RDFs) from a given XYZ file.

import torch

from PyISV.utils.define_root import PROJECT_ROOT as root_dir
from PyISV.utils.features_calc_utils import build_rdf

# Define global constants for testing purposes
XYZ_PATH = f"{root_dir}/data/structures/full_min_ptmd_m18_nCu_0.xyz"
OUTPUT_DIR = f"{root_dir}/data/RDFs"
MIN_DIST = 1.0
MAX_DIST = 12.0
N_BINS = 340
BANDWIDTH = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
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
        print(f"Error during RDF computation: {e}")
    finally:
        print("RDF computation finished.")

